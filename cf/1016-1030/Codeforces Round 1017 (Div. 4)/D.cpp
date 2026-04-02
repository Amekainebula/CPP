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
#define vi vector<int>
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
void Murasame()
{
    string s, t;
    cin >> s >> t;
    int nows = 0, nowt = 0;
    int type = -1;
    while (nows < s.size() && nowt < t.size())
    {
        if (s[nows] == t[nowt])
        {
            type = s[nows];
            int cnts = 0, cntt = 0;
            while (nows < s.size() && s[nows] == type)
            {
                cnts++;
                nows++;
            }
            while (nowt < t.size() && t[nowt] == type)
            {
                cntt++;
                nowt++;
            }
            if (!(cntt >= cnts && cntt <= cnts * 2))
            {
                cout << "No" << endl;
                return;
            }
        }
        else
        {
            cout << "No" << endl;
            return;
        }
    }
    if (nows == s.size() && nowt == t.size())
        cout << "Yes" << endl;
    else
        cout << "No" << endl;
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