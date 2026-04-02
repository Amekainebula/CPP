#include <bits/stdc++.h>
#define int long long
#define ull unsigned long long
#define i128 __int128
#define ld long double
#define ff(x, y, z) for (int(x) = (y); (x) <= (z); ++(x))
#define ffg(x, y, z) for (int(x) = (y); (x) >= (z); --(x))
#define pb push_back
#define eb emplace_back
#define pii pair<int, int>
#define vc vector
#define vi vector<int>
#define vvi vector<vi>
#define fi first
#define se second
#define all(x) x.begin(), x.end()
#define all1(x) x.begin() + 1, x.end()
#define INF 0x7fffffffffffffff
#define inf 0x7fffffff
// #define endl endl << flush
#define endl '\n'
#define WA AC
#define TLE AC
#define MLE AC
#define RE AC
#define CE AC
using namespace std;
const string AC = "Accepted";
const int MOD = 1e9 + 7;
const int mod = 998244353;
const int N = 5e5 + 5;
vi a(N);
void Murasame()
{
    int n;
    cin >> n;
    int temp = 0;
    bool ok = 1;
    ff(i, 1, n)
    {
        int x;
        cin >> x;
        a[i] = x;
        if (i == 1)
        {
            temp = x;
        }
        else
        {
            if (x != temp)
            {
                ok = 0;
            }
        }
    }
    if (ok)
    {
        cout << 0 << endl;
        return;
    }
    int ans = INF;
    ff(i, 1, n)
    {
        int cnt = 1;
        ff(j, i + 1, n)
        {
            if (a[i] == a[j])
                cnt++;
            else
                break;
        }
        ans = min(ans, (n - cnt) * a[i]);
        i += cnt - 1;
    }
    cout << ans << endl;
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
        Murasame();
    }
    return 0;
}