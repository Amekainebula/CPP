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
#define fi first
#define se second
#define all(x) x.begin(), x.end()
#define all1(x) x.begin() + 1, x.end()
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
void Murasame()
{
    int n;
    cin >> n;
    vi a(n + 1);
    map<int, int> mp;
    ff(i, 1, n)
    {
        cin >> a[i];
        mp[a[i]]++;
    }
    map<int, int> v;
    for (auto [x, cnt] : mp)
    {
        if (cnt >= 4)
        {
            cout << "YES" << endl;
            return;
        }
        if (v[x + 1] && cnt >= 2)
        {
            cout << "YES" << endl;
            return;
        }
        if (v[x + 1] || cnt >= 2)
            v[x + 2] = 1;
    }
    cout << "NO" << endl;
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