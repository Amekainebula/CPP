#include <bits/stdc++.h>
#define int long long
#define ull unsigned long long
#define i128 __int128
#define ld long double
#define ff(x, y, z) for (int(x) = (y); (x) <= (z); ++(x))
#define ffg(x, y, z) for (int(x) = (y); (x) >= (z); --(x))
#define pb push_back
#define eb emplace_back
#define vc vector
#define pii pair<int, int>
#define mpr make_pair
#define fi first
#define se second
#define all(x) x.begin(), x.end()
#define INF 0x7fffffffffffffff
#define inf 0x7fffffff
#define endl '\n'
#define WA AC
#define TLE AC
#define MLE AC
#define RE AC
#define CE AC
using namespace std;
const string AC = "Accepted";
void solve()
{
    int n;
    cin >> n;
    vector<int> a(n), b(n);
    int suma = 0, sumb = 0;
    ff(i, 0, n - 1)
    {
        cin >> a[i];
        suma += a[i];
    }
    ff(i, 0, n - 1)
    {
        cin >> b[i];
        sumb += b[i];
    }
    sort(all(a));
    sort(all(b));
    int temp = suma - sumb;
    if (temp < 0)
    {
        cout << -1 << endl;
        return;
    }
    if (temp == 0)
    {
        cout << (a == b ? 1000000000 : -1) << endl;
        return;
    }
    auto check = [=](int x)
    {
        vc<int> tp;
        for (int i = 0; i < n; i++)
        {
            tp.pb(a[i] % x);
        }
        sort(all(tp));
        if (tp == b)
            return true;
        return false;
    };
    for (int i = 1; i * i <= temp; i++)
    {
        if (temp % i == 0)
        {
            if (check(i))
            {
                cout << i << endl;
                return;
            }
            if (check(temp / i))
            {
                cout << temp / i << endl;
                return;
            }
        }
    }
    cout << -1 << endl;
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int _T = 1;
    //
    cin >> _T;
    while (_T--)
    {
        solve();
    }
    return 0;
}