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
void Murasame()
{
    int n;
    cin >> n;
    vc<int> a(n + 1);
    map<int, int> mp;
    ff(i, 1, n)
    {
        int x;
        cin >> x;
        mp[x]++;
        a[i] = x;
    }
    sort(all1(a));
    int ans = 1;
    ff(i, 1, n) ff(j, i + 1, n)
    {
        int temp = a[i] + a[j];
        if (temp % 2 || mp[temp / 2] == 0 || j - i + 1 < ans)
            continue;
        if (a[i] == a[j])
        {
            ans = max(ans, j - i + 1);
            continue;
        }
        if (ans == n)
        {
            cout << ans << endl;
            return;
        }
        vc<int> q;
        ff(k, i, j)
        {
            q.push_back(a[k]);
        }
        while (q[(q.size() + 1) / 2 - 1] != temp / 2)
        {
            q.erase(q.begin() + (q.size() + 1) / 2 - 1);
        }
        ans = max(ans, (int)q.size());
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